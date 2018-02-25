//
//  ViewController.swift
//  CoreMLTest
//
//  Created by 藤井孝治 on 2018/02/24.
//  Copyright © 2018年 gadget-tf. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, ClassificationDelegate {
    
    func didClassify(with identifier: String) {
        DispatchQueue.main.async {
            self.label.text = identifier
        }
        print(identifier)
    }
    
    @IBOutlet weak var label: UILabel!
    var modelData: VNCoreMLModel?
    var delegate: ClassificationDelegate?

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        delegate = self
        do {
            modelData = try VNCoreMLModel(for: flower().model)
        } catch {
            fatalError()
        }
        //let image = UIImage(named: "1.jpg")
        //initML(image: image!)
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    func initML(image: UIImage) {
        guard let ciImage = CIImage(image: image) else {
            fatalError()
        }
        
        let request = VNCoreMLRequest(model: modelData!) {(result, error) in
            if let err = error {
                print(err)
            }
            guard let result = result.results as? [VNClassificationObservation] else {
                fatalError()
            }
            guard let best = result.first else {
                fatalError()
            }
            self.delegate?.didClassify(with: best.identifier)
            print(best.identifier)
            print(best.confidence * 100)
        }
        
        let handler = VNImageRequestHandler(ciImage: ciImage)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                print("error")
            }
        }
    }

    @IBAction func banana(_ sender: Any) {
        initML(image: UIImage(named: "1.jpg")!)
    }
    
    @IBAction func apple(_ sender: Any) {
        initML(image: UIImage(named: "7.jpg")!)
    }
    
    @IBAction func grape(_ sender: Any) {
        initML(image: UIImage(named: "6.jpg")!)
    }
    
}

